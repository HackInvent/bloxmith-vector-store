"""FB8: real release UI in managed and linked hosts: drafts, server validation, EN/FR, responsive geometry."""

from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError, expect, sync_playwright
SETTLE_MS = 15000  # Shared test hosts can be slow; assertions still fail on real defects.
expect.set_options(timeout=SETTLE_MS)
from block_test_artifacts import artifact_path
from block_test_packages import install_test_package, surface_payload
from ui_smoke_common import isolated_server, graph_payload, create_project_api, project_editor_url
from blocs.vector_store.block import VectorStoreBlock

MEASURE = """(el, bodySelector) => {
    const r=el.getBoundingClientRect(), body=el.querySelector(bodySelector) || el;
    const actions=[...el.querySelectorAll('[data-close-block-modal], [data-owned-apply]')];
    return {inside:r.left>=-1&&r.right<=innerWidth+1&&r.top>=-1&&r.bottom<=innerHeight+1,
      overflow:el.scrollWidth>el.clientWidth+2||body.scrollWidth>body.clientWidth+2,
      controls:actions.every(a=>{const b=a.getBoundingClientRect();return b.top>=0&&b.bottom<=innerHeight+1}),
      background:getComputedStyle(el).backgroundColor,
      unlabeled:[...el.querySelectorAll('input,select,textarea')].filter(c=>!c.labels?.length&&!c.getAttribute('aria-label')&&!c.getAttribute('aria-labelledby')).length};
}"""


def geometry_ok(measured):
    return (measured["inside"] and measured["controls"] and not measured["overflow"] and not measured["unlabeled"]
            and measured["background"] not in {"transparent", "rgba(0, 0, 0, 0)"})


def resize(page, width, height):
    """Resize, then wait until the page reports the new viewport and has painted twice."""
    page.set_viewport_size({"width": width, "height": height})
    page.wait_for_function("([w, h]) => innerWidth === w && innerHeight === h", arg=[width, height], timeout=SETTLE_MS)
    page.evaluate("() => new Promise(done => requestAnimationFrame(() => requestAnimationFrame(done)))")


def geometry(surface, body=".owned-body"):
    """Opaque panel, reachable actions, labelled controls and no horizontal overflow (polled while the editor re-lays out)."""
    deadline = time.monotonic() + SETTLE_MS / 1000
    measured = surface.evaluate(MEASURE, body)
    while not geometry_ok(measured) and time.monotonic() < deadline:
        surface.page.wait_for_timeout(100)
        measured = surface.evaluate(MEASURE, body)
    assert measured["inside"] and measured["controls"] and not measured["overflow"], measured
    assert not measured["unlabeled"], measured
    assert measured["background"] not in {"transparent", "rgba(0, 0, 0, 0)"}, measured


def field(surface, key):
    return surface.locator(f'[data-block-config-field="{key}"]')


def open_modal(card, modal):
    """Open the modal from the canvas card; after a reload the editor may bind its handlers late."""
    expect(card.locator('h3')).to_be_visible(timeout=30000)
    for _ in range(6):
        if modal.is_visible():
            return
        try:
            card.locator('h3').dblclick(timeout=5000)
            expect(modal).to_be_visible(timeout=5000)
            return
        except (AssertionError, PlaywrightTimeoutError):
            continue
    expect(modal).to_be_visible()


def saving(response):
    """Match the Apply request itself, not another surface action posted meanwhile."""
    return (response.url.endswith('/ui-action') and response.request.method == 'POST'
            and 'save_settings' in (response.request.post_data or ''))


def open_advanced(modal):
    details = modal.locator('.owned-advanced').first
    if details.get_attribute('open') is None:
        details.locator('summary').click()
    expect(details).to_have_attribute('open', '')


def main():
    """Exercise the current package files through the real managed and linked browser hosts."""
    for origin in ("managed", "linked"):
        with isolated_server() as server, sync_playwright() as playwright:
            model = install_test_package(server, "vector_store", origin=origin)
            candidate = VectorStoreBlock().build_node_payload(node_id="ui-case", position={"x": 160, "y": 160})
            candidate["block_version"] = model["version"]
            # - FB8 - Every surface declares its assets and a complete EN/FR catalog for its markers.
            for surface in ("modal", "inspector_panel", "node_card", "mini_node_card"):
                payload = surface_payload(server, model, candidate, surface=surface)
                assert "{{" not in payload["html"] and set(payload["locales"]) >= {"en", "fr"}
            project = create_project_api(server, document=graph_payload("Vector settings", [candidate], []))["project"]
            url = project_editor_url(server.base_url, project["project_id"], workspace_project_id=project["workspace_project_id"])
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(viewport={"width": 1440, "height": 900})
                page.add_init_script("window.localStorage.setItem('bloxsmith.inspectorPinned','true')")
                errors = []
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.goto(url)
                card = page.locator('.canvas-node[data-node-id="ui-case"]')
                expect(card.locator('h3')).to_be_visible()
                expect(card.locator('.node-data-preview')).to_contain_text("default · nomic-embed-text")
                modal = page.locator('.owned-modal')
                apply = modal.locator('[data-owned-apply]')
                open_modal(card, modal)
                expect(apply).to_be_disabled()
                expect(field(modal, "base_url")).to_have_value("http://127.0.0.1:11434/v1")
                expect(field(modal, "allow_delete")).not_to_be_checked()
                # Cancel discards the draft.
                field(modal, "model").fill("draft-model")
                expect(apply).to_be_enabled()
                modal.locator('[data-close-block-modal]').last.click()
                open_modal(card, modal)
                expect(field(modal, "model")).to_have_value("nomic-embed-text")
                # A pasted key instead of a wallet reference is refused by the server and nothing is persisted.
                field(modal, "api_key_ref").fill("sk-pasted-key-0000")
                with page.expect_response(saving, timeout=60000):
                    apply.click()
                expect(modal.locator('.owned-form-status')).to_contain_text("secret://workspace/", timeout=60000)
                modal.locator('[data-close-block-modal]').first.click()
                page.reload()
                open_modal(card, modal)
                expect(field(modal, "api_key_ref")).to_have_value("")
                # A valid draft is applied and survives a reload, including a prefix ending with a space.
                field(modal, "model").fill("bge-m3")
                field(modal, "top_k").fill("8")
                field(modal, "min_score").fill("0.25")
                field(modal, "allow_delete").check()
                open_advanced(modal)
                field(modal, "query_prefix").fill("search_query: ")
                with page.expect_response(saving, timeout=60000) as saved:
                    apply.click()
                assert not saved.value.json().get("error"), saved.value.json()
                expect(modal.locator('.owned-form-status')).to_contain_text("Settings applied")
                if modal.is_visible():
                    modal.locator('[data-close-block-modal]').first.click()
                page.reload()
                expect(card.locator('.node-data-preview')).to_contain_text("default · bge-m3")
                open_modal(card, modal)
                expect(field(modal, "model")).to_have_value("bge-m3")
                expect(field(modal, "top_k")).to_have_value("8")
                expect(field(modal, "min_score")).to_have_value("0.25")
                expect(field(modal, "allow_delete")).to_be_checked()
                expect(field(modal, "query_prefix")).to_have_value("search_query: ")
                for width, height in ((1440, 900), (800, 700), (390, 740), (320, 568)):
                    resize(page, width, height)
                    geometry(modal)
                    page.screenshot(path=artifact_path(f"vector_store-{origin}-{width}.png"))
                summary = modal.locator('.owned-advanced').first.locator('summary')
                summary.scroll_into_view_if_needed()
                if modal.locator('.owned-advanced').first.get_attribute('open') is not None:
                    summary.click()
                summary.focus()
                summary.press('Enter')
                expect(modal.locator('.owned-advanced').first).to_have_attribute('open', '')
                expect(field(modal, "max_database_mb")).to_be_visible()
                geometry(modal)
                page.screenshot(path=artifact_path(f"vector_store-{origin}-advanced.png"))
                resize(page, 1440, 900)
                modal.locator('[data-close-block-modal]').first.click()
                # The inspector exposes the same fields and the Ports tab.
                card.click(position={"x": 20, "y": 20})
                inspector = page.locator('[data-properties-surface="inspector"][data-node-id="ui-case"]:visible')
                expect(inspector).to_be_visible()
                assert inspector.evaluate('el => el.scrollWidth <= el.clientWidth+2')
                expect(field(inspector, "model")).to_have_value("bge-m3")
                field(inspector, "collection").fill("manuals")
                expect(inspector.locator('[data-owned-apply]')).to_be_enabled()
                inspector.locator('[data-inspector-tab="ports"]').click()
                expect(inspector.locator('[data-inspector-panel-tab="ports"]')).to_be_visible()
                expect(inspector.locator('[data-inspector-panel-tab="ports"]')).to_contain_text("documents")
                inspector.locator('[data-inspector-tab="general"]').click()
                expect(inspector.locator('[data-inspector-panel-tab="general"]')).to_be_visible()
                page.screenshot(path=artifact_path(f"vector_store-{origin}-inspector.png"))
                for width, height in ((390, 740), (320, 568)):
                    resize(page, width, height)
                    page.screenshot(path=artifact_path(f"vector_store-{origin}-inspector-{width}.png"))
                # The reduced display uses the block-owned mini card, not an error placeholder.
                page.goto(url)
                resize(page, 390, 740)
                page.locator('#compactView [data-compact-project-id]').first.click()
                mini = page.locator('#compactView [data-vector-store-mini-card]').first
                expect(mini).to_be_visible()
                expect(mini).to_contain_text("default · bge-m3")
                page.screenshot(path=artifact_path(f"vector_store-{origin}-compact-390.png"))
                resize(page, 1440, 900)
                # French interface: block texts come from the package catalog.
                page.goto(server.base_url + "/")
                page.locator("#homeApplicationSettingsButton").click()
                page.locator("#applicationLanguageSelect").select_option("fr")
                page.wait_for_function("window.CWMessages.getLanguage() === 'fr'")
                page.goto(url)
                open_modal(card, modal)
                expect(modal.locator('.owned-advanced').first.locator('summary')).to_have_text("Réglages avancés")
                expect(modal.locator('.panel-kicker')).to_have_text("Recherche sémantique")
                expect(modal.locator(f'#{field(modal, "collection").get_attribute("aria-describedby")}')).to_contain_text("stockage de ce nœud")
                resize(page, 390, 740)
                geometry(modal)
                page.screenshot(path=artifact_path(f"vector_store-{origin}-fr-390.png"))
                assert not errors, errors
                assert page.evaluate("!window.CWBlockUiBlocks?.vector_store")
            finally:
                browser.close()
        print("[ok] vector_store " + origin + " UI", flush=True)


if __name__ == "__main__":
    main()
