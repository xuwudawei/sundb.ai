let BASE_URL: string;

const script = document.currentScript as HTMLScriptElement | null;
if (!script) {
  throw new Error(`Widget not supported in this browser (evaluating document.currentScript)`);
}
// data-api-base
if (script.dataset.apiBase) {
  BASE_URL = script.dataset.apiBase;
  console.debug('[tidbai.widget]', 'widget base url resolved by "data-api-base" attribute', BASE_URL);
} else if (/^https?:\/\//.test(script.src)) {
  const scriptUrl = new URL(script.src);
  BASE_URL = scriptUrl.origin;
  console.debug('[tidbai.widget]', 'widget base url resolved by script origin', BASE_URL);
} else {
  console.warn(`Add attribute "data-api-base"="YOUR_HOST" to your widget script tag.`);
  throw new Error(`Cannot initialize widget.`);
}

export { BASE_URL }