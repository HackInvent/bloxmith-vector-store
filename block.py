"""Embed texts through an OpenAI-compatible endpoint and search them by meaning in a node-private collection."""

import json
import sqlite3

from bloxsmith_app.block_api import (APPLICATION_JSON, BlockDefinition, BlockRuntimeOutput,
                                     BlockRuntimePreparation, BlockRuntimeResult)
from . import engine, logic, ui
from .logic import VectorError

PORTS = ("documents", "query")


# FB1 - Validate settings, endpoint policy and every message (texts, objects, envelopes, filters) before any call or write.
# FB2 - Chunk texts by characters with overlap and derive deterministic ids; unchanged chunks are never re-embedded.
# FB3 - Bind each collection to one model and dimension; refuse mismatches, bound capacity and write each message atomically.
# FB4 - Rank by exact cosine similarity with top_k, min_score and metadata equality filters; numpy optional with a Python fallback.
# FB5 - Delete by ids, filter or all only when allow_delete is enabled; report stats without calling the provider.
# FB6 - Resolve the API key only from the wallet, contact loopback unless allow_remote, never leak secrets or provider bodies.
# FB7 - Publish matches/result/error identically in both runtimes and installations; Stop aborts a pending provider call.
# FB8 - Provide translated responsive settings with draft/Apply validation and release-scoped assets.
class VectorStoreBlock(BlockDefinition):
    """Retrieval returns stored passages and scores; it never generates or verifies an answer."""

    kind = "vector_store"

    def ui_assets(self, surface="modal"):
        """Every installation origin uses the static release manifest."""
        return list(self.model.get("ui_assets", {}).get(surface, []))

    def prepare_runtime(self, context):
        """Validate settings without network, wallet or storage access."""
        logic.configuration(context.config)
        return BlockRuntimePreparation()

    def initialize_runtime(self, context):
        """Fail the Run early when this node has no persistent storage."""
        try:
            logic.configuration(context.config)
            engine.Services({}, storage=context.services.get("get_block_storage_dir")).directory()
        except VectorError as exc:
            return self._failed(str(exc))
        return BlockRuntimeResult(status="success")

    def execute_runtime(self, context):
        """Mutations run before queries; each message succeeds or publishes one error."""
        try:
            config = logic.configuration(context.config)
        except VectorError as exc:
            return self._failed(str(exc))
        messages, errors = self._messages(context)
        if not messages and not errors:
            return BlockRuntimeResult(status="skipped", outputs=[])
        services = engine.Services(config, storage=context.services.get("get_block_storage_dir"),
                                   resolve_secret=lambda reference: context.resolve_secret(reference),
                                   cancel=context.services.get("cancel_requested", lambda: False))
        values, done = {}, []
        for port, value in messages:
            try:
                produced = engine.process(value, port, config, services)
                values.update(produced)
                done.append(next(iter(produced.values())).get("operation", "query"))
            except VectorError as exc:
                errors.append(exc.envelope(getattr(exc, "operation", ""), port))
            except (ValueError, TypeError, OSError, RuntimeError, sqlite3.Error, RecursionError, MemoryError):
                # Raw driver, path, provider or credential details never reach outputs.
                errors.append(VectorError("internal", "The operation ended without a complete result; nothing was stored.")
                              .envelope("", port))
        if errors:
            values["error"] = {**errors[0], **({"additional_errors": errors[1:]} if len(errors) > 1 else {})}
        outputs = []
        for port in context.output_ports:
            if port.name in values:
                text = json.dumps(values[port.name], ensure_ascii=False, allow_nan=False)
                outputs.append(BlockRuntimeOutput(port_id=int(port.id), port_name=port.name,
                                                  value=context.redact_secrets(text), content_type=APPLICATION_JSON))
        if errors:
            log = f"[vector-store] {config['collection']}: {len(done)} operation(s) done, failed with {errors[0]['code']}."
        else:
            log = f"[vector-store] {config['collection']}: {', '.join(done)} done."
        found = values.get("matches") or {}
        return BlockRuntimeResult(
            status="success", outputs=outputs, content_type=APPLICATION_JSON,
            last_message=json.dumps({"code": errors[0]["code"]} if errors else {"operations": done}),
            worker_received=str(len(messages)),
            metadata={self.kind: {"status": "error" if errors else "ok", "collection": config["collection"],
                                  "operations": done, "matches": found.get("count", 0),
                                  "code": errors[0]["code"] if errors else ""}},
            logs=[context.redact_secrets(log)])

    @staticmethod
    def _messages(context):
        """At most one message per input in one activation; documents before query."""
        found, errors = {}, []
        if context.input_events:
            names = {int(port.id): str(port.name) for port in context.input_ports}
            for event in context.input_events:
                name = names.get(int(event.input_port_id))
                if name not in PORTS:
                    errors.append(VectorError("input", "Unknown input port; the message was ignored.").envelope())
                    continue
                found.setdefault(name, []).append(event.value)
        else:
            for ident, name in enumerate(PORTS, 1):
                if context.has_input_value(name, str(ident)):
                    found[name] = [context.input_value(name, str(ident))]
        messages = []
        for name in PORTS:
            items = found.get(name, [])
            if len(items) > 1:
                errors.append(VectorError("batch", f"{len(items)} messages reached {name} in one activation; "
                                          "none of them was processed.").envelope("", name))
            elif items:
                messages.append((name, items[0]))
        return messages, errors

    def _failed(self, detail):
        message = self.translate("block.vector_store.error", {"detail": detail}, fallback="Vector Store failed: {detail}")
        return BlockRuntimeResult(status="failed", outputs=[], error=message, exit_code=1,
                                  logs=[f"[vector-store-error] {detail}"])

    def handle_ui_action(self, *, node, action, values, payload=None):
        """Apply validates settings; it never contacts the endpoint, opens the collection or reads the wallet."""
        if action in {"save_settings", "modal_update_fields", "inspector_update_fields"}:
            patch = (values or {}).get("node_patch") or {}
            if "config" in patch:
                try:
                    logic.configuration({**self.default_config(), **(node.get("config") or {}), **patch["config"]})
                except (VectorError, ValueError, TypeError) as exc:
                    detail = str(exc) if isinstance(exc, VectorError) else "Invalid Vector Store settings."
                    return {"error": self.translate("block.vector_store.error", {"detail": detail},
                                                    fallback="Vector Store failed: {detail}")}
        result = super().handle_ui_action(node=node, action="modal_update_fields" if action == "save_settings" else action,
                                          values=values, payload=payload)
        return ui.replace_config(result, node)

    def render_modal(self, *, node, payload=None):
        """Endpoint, model and search settings stay a draft until Apply."""
        return ui.modal(self, node, payload)

    def render_inspector_panel(self, *, node, payload=None):
        """The same validated fields in the narrow inspector."""
        return ui.inspector(self, node, payload)

    def render_node_card(self, *, node, payload=None):
        """Show the collection and model only, never the endpoint, the wallet reference or stored text."""
        return ui.card(self, node, self._preview(node))

    def render_mini_node_card(self, *, node, payload=None):
        """Compact display card with the same safe preview as the canvas card."""
        return ui.mini_card(self, node, self._preview(node))

    def _preview(self, node):
        config = {**self.default_config(), **(node.get("config") or {})}
        return f"{config.get('collection')} · {config.get('model')}"
