/** Bind release-owned settings through the public action API. */
import { enhanceProperties } from "./properties.js";
import { bindSettings } from "./settings.js";
export function mount(root, api) {
  const disposeStyle = enhanceProperties(root), disposeForm = bindSettings(root, api);
  return {dispose() {disposeForm(); disposeStyle();}};
}
