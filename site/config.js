// Where the backend lives.
//
// The models run on a Mac; this page is static and hosted anywhere. That split
// is the whole architecture, so the address of the machine is the one piece of
// configuration this site has.
//
// Cloudflare's *quick* tunnels hand out a new hostname every restart, so expect
// to edit this line. `?api=https://...` overrides it without a redeploy, which
// is how to check a new tunnel before committing to it.
window.SWAR_API =
  new URLSearchParams(location.search).get("api") ||
  "https://surrounding-mil-serving-subdivision.trycloudflare.com";
