// "Identify from a link": an iNaturalist or Mushroom Observer observation, as someone
// pastes it, read into what POST /api/identify/observation takes.

export type ObservationSource = "inat" | "mo";
export interface ObservationRef { source: ObservationSource; id: string }

const ID = /^\d{1,12}$/;

/** iNaturalist and the sites of its network; an observation number is the same on all. */
const INAT_DOMAINS = [
  "inaturalist.org", "inaturalist.ca", "inaturalist.nz", "naturewatch.org.nz", "inaturalist.lu",
  "inaturalist.se", "inaturalist.ala.org.au", "naturalista.mx", "naturalista.uy", "argentinat.org",
  "biodiversity4all.org", "inaturalist.mma.gob.cl", "israelinat.org.il", "inaturalist.laji.fi",
];

function isInatHost(host: string): boolean {
  return INAT_DOMAINS.some((d) => host === d || host.endsWith(`.${d}`));
}

function isMoHost(host: string): boolean {
  return host === "mushroomobserver.org" || host.endsWith(".mushroomobserver.org");
}

/**
 * The observation a link points at, or null. Takes iNaturalist's
 * /observations/<n> (any site of the network, the app's share links too) and Mushroom
 * Observer's /<n>, /obs/<n>, /observations/<n> and /observer/show_observation/<n>,
 * with or without https:// and www. Also "inat:<n>" and "mo:<n>". A bare number
 * says neither site, so it is not taken.
 */
export function parseObservationLink(text: string): ObservationRef | null {
  const raw = (text ?? "").trim();
  if (!raw) return null;
  const short = raw.match(/^(inat|mo)\s*[:#]\s*(\d{1,12})$/i);
  if (short) return { source: short[1].toLowerCase() as ObservationSource, id: String(Number(short[2])) };
  let url: URL;
  try {
    url = new URL(/^[a-z][a-z0-9+.-]*:\/\//i.test(raw) ? raw : `https://${raw}`);
  } catch {
    return null;
  }
  const host = url.hostname.toLowerCase().replace(/^www\./, "");
  const parts = url.pathname.split("/").filter(Boolean);
  if (isInatHost(host)) {
    const at = parts.indexOf("observations");
    const id = at >= 0 ? parts[at + 1] : undefined;
    return id && ID.test(id) && Number(id) > 0 ? { source: "inat", id: String(Number(id)) } : null;
  }
  if (isMoHost(host)) {
    let id: string | undefined;
    if (parts.length === 1) id = parts[0];
    else if (parts.length === 2 && (parts[0] === "obs" || parts[0] === "observations")) id = parts[1];
    else if (parts.length === 3 && parts[0] === "observer" && parts[1] === "show_observation") id = parts[2];
    return id && ID.test(id) && Number(id) > 0 ? { source: "mo", id: String(Number(id)) } : null;
  }
  return null;
}

export const sourceName = (s: ObservationSource) => (s === "inat" ? "iNaturalist" : "Mushroom Observer");
