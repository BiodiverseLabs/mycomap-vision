// The map for picking where a find was (Steve, 2026-10-11), loaded only when someone opens
// it (React.lazy in pages/Identify.tsx), so Leaflet stays out of the page's first load.
// Full screen on a phone, a dialog on a computer. Tiles as on mycomap.org's maps; place
// search through Vision's /api/geocode/search (geocode.py). The rules are lib/mapPick.ts.
import { useEffect, useId, useRef, useState, type FormEvent } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { MapContainer, Marker, TileLayer, useMap, useMapEvents } from "react-leaflet";
import { Crosshair, Layers, Loader2, LocateFixed, Search, X } from "lucide-react";
import { Button } from "@/components/ui/button";
import type { Where } from "@/lib/api";
import { fieldsFromPoint, PLACE_ZOOM, pointFromFields, startView, type Point } from "@/lib/mapPick";
import { locateError } from "@/lib/photoPlace";

const BASEMAPS = {
  map: {
    label: "Map",
    url: "https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
    attribution: 'Map data &copy; <a href="https://openstreetmap.org">OpenStreetMap</a> contributors, SRTM | Map style &copy; <a href="https://opentopomap.org">OpenTopoMap</a>',
    maxZoom: 17,
  },
  satellite: {
    label: "Satellite",
    url: "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attribution: "Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics",
    maxZoom: 19,
  },
} as const;
type Basemap = keyof typeof BASEMAPS;

// A drawn pin: no marker images to load (and none for the site's image policy to allow).
const PIN = L.divIcon({
  className: "",
  html: '<svg width="32" height="40" viewBox="0 0 24 30" aria-hidden="true"><path d="M12 29s9-9.3 9-16a9 9 0 1 0-18 0c0 6.7 9 16 9 16z" fill="#8bc34a" stroke="#3f5f1a" stroke-width="1.5"/><circle cx="12" cy="12" r="3.5" fill="#fff"/></svg>',
  iconSize: [32, 40],
  iconAnchor: [16, 40],
});

interface Place { display_name: string; lat: number; lon: number }

function ClickToPin({ onPick }: { onPick: (p: Point) => void }) {
  useMapEvents({ click: (e) => onPick({ lat: e.latlng.lat, lng: e.latlng.lng }) });
  return null;
}

/** Moves the map when `target` changes (a search result, "Centre on me", typed fields). */
function FlyTo({ target }: { target: { p: Point; zoom: number; n: number } | null }) {
  const map = useMap();
  useEffect(() => {
    if (target) map.setView([target.p.lat, target.p.lng], Math.max(map.getZoom(), target.zoom));
  }, [target, map]);
  return null;
}

export default function LocationPicker({ where, onUse, onClose }: {
  where: Where;
  onUse: (pin: Point) => void;
  onClose: () => void;
}) {
  const start = useRef(startView(where)).current;
  const [pin, setPin] = useState<Point | null>(start.pin);
  const [fields, setFields] = useState(start.pin ? fieldsFromPoint(start.pin) : { lat: "", lng: "" });
  const [basemap, setBasemap] = useState<Basemap>("map");
  const [fly, setFly] = useState<{ p: Point; zoom: number; n: number } | null>(null);
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<Place[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [locating, setLocating] = useState(false);
  const map = useRef<L.Map | null>(null);
  const titleId = useId();
  const searchBox = useRef<HTMLInputElement>(null);

  // Pin → fields.
  const placePin = (p: Point, flyThere = false) => {
    setPin(p);
    setFields(fieldsFromPoint(p));
    if (flyThere) setFly({ p, zoom: PLACE_ZOOM, n: Date.now() });
  };
  // Fields → pin, once they make a real point.
  const typeField = (k: "lat" | "lng", v: string) => {
    const next = { ...fields, [k]: v };
    setFields(next);
    const p = pointFromFields(next.lat, next.lng);
    if (p) {
      setPin(p);
      setFly({ p, zoom: map.current?.getZoom() ?? PLACE_ZOOM, n: Date.now() });
    }
  };

  // A sheet over the page: Escape closes it, the page behind doesn't scroll.
  useEffect(() => {
    const before = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", onKey);
    searchBox.current?.focus();
    return () => {
      document.body.style.overflow = before;
      window.removeEventListener("keydown", onKey);
    };
  }, [onClose]);

  const search = async (e: FormEvent) => {
    e.preventDefault();
    if (query.trim().length < 3) return setMessage("Type at least 3 letters of a place name.");
    setSearching(true);
    setMessage(null);
    try {
      const res = await fetch(`/api/geocode/search?q=${encodeURIComponent(query.trim())}`);
      const body = await res.json().catch(() => null);
      if (!res.ok) throw new Error(body?.detail ?? "Place search isn't answering; try again in a minute.");
      setResults(body as Place[]);
      if (!(body as Place[]).length) setMessage("No place by that name. Try a nearby town or park.");
    } catch (err) {
      setResults(null);
      setMessage((err as Error).message);
    } finally {
      setSearching(false);
    }
  };

  const centreOnMe = () => {
    if (!navigator.geolocation) return setMessage(locateError(2));
    setLocating(true);
    setMessage(null);
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setLocating(false);
        placePin({ lat: pos.coords.latitude, lng: pos.coords.longitude }, true);
      },
      (err) => {
        setLocating(false);
        setMessage(locateError(err.code));
      },
      { timeout: 15_000, maximumAge: 60_000 },
    );
  };

  const pinAtCentre = () => {
    const c = map.current?.getCenter();
    if (c) placePin({ lat: c.lat, lng: c.lng });
  };

  const base = BASEMAPS[basemap];
  const input = "h-10 w-full min-w-0 rounded-md border border-input bg-background px-2 text-sm";
  return (
    <div className="fixed inset-0 z-[60] flex items-stretch justify-center bg-black/40 lg:items-center lg:p-6"
         data-testid="sheet-location-picker">
      <div role="dialog" aria-modal="true" aria-labelledby={titleId}
           className="flex h-full w-full flex-col bg-white lg:h-[85vh] lg:max-w-4xl lg:rounded-lg lg:shadow-xl">
        <div className="flex items-center justify-between gap-2 border-b px-4 py-3">
          <h2 id={titleId} className="font-display font-semibold text-lg text-[#4a3728]">Pick where you found it</h2>
          <Button type="button" variant="ghost" size="icon" onClick={onClose} aria-label="Close the map">
            <X />
          </Button>
        </div>

        <form onSubmit={search} className="flex gap-2 px-4 pt-3" role="search">
          <input ref={searchBox} className={input} value={query} onChange={(e) => setQuery(e.target.value)}
                 placeholder="Search for a place, park or town" aria-label="Search for a place"
                 data-testid="input-place-search" />
          <Button type="submit" variant="outline" className="h-10 shrink-0" disabled={searching}>
            {searching ? <Loader2 className="animate-spin" /> : <Search />}
            <span className="sr-only sm:not-sr-only">Search</span>
          </Button>
        </form>
        {results && results.length > 0 && (
          <ul className="mx-4 mt-2 max-h-40 overflow-y-auto rounded-md border text-sm" data-testid="list-place-results">
            {results.map((r) => (
              <li key={`${r.lat},${r.lon}`}>
                <button type="button" className="w-full px-3 py-2 text-left hover:bg-muted focus-visible:bg-muted"
                        onClick={() => { placePin({ lat: r.lat, lng: r.lon }, true); setResults(null); }}>
                  {r.display_name}
                </button>
              </li>
            ))}
          </ul>
        )}
        {message && <p className="mx-4 mt-2 text-xs text-destructive" role="status">{message}</p>}

        <div className="relative mt-3 min-h-[260px] flex-1">
          <MapContainer ref={map} center={[start.center.lat, start.center.lng]} zoom={start.zoom}
                        className="h-full w-full" keyboard worldCopyJump>
            <TileLayer key={basemap} url={base.url} attribution={base.attribution} maxZoom={base.maxZoom} />
            {pin && (
              <Marker position={[pin.lat, pin.lng]} icon={PIN} draggable keyboard={false}
                      eventHandlers={{ dragend: (e) => placePin((e.target as L.Marker).getLatLng()) }} />
            )}
            <ClickToPin onPick={(p) => placePin(p)} />
            <FlyTo target={fly} />
          </MapContainer>
          <div className="absolute right-2 top-2 z-[400] flex flex-col gap-2">
            <Button type="button" variant="outline" size="sm" className="bg-white shadow"
                    onClick={() => setBasemap(basemap === "map" ? "satellite" : "map")}
                    aria-label={`Show ${basemap === "map" ? "satellite" : "map"} view`}>
              <Layers /> {basemap === "map" ? "Satellite" : "Map"}
            </Button>
            <Button type="button" variant="outline" size="sm" className="bg-white shadow" onClick={centreOnMe}
                    disabled={locating}>
              {locating ? <Loader2 className="animate-spin" /> : <LocateFixed />} Centre on me
            </Button>
          </div>
        </div>

        <div className="space-y-2 border-t px-4 py-3 pb-[calc(0.75rem+env(safe-area-inset-bottom))]">
          <p className="text-xs text-muted-foreground">
            Tap the map or drag the pin. With a keyboard, move the map with the arrow keys, then
            put the pin at its centre.
          </p>
          <div className="flex flex-wrap items-center gap-2">
            <input className={`${input} flex-1 basis-28`} inputMode="decimal" placeholder="Latitude"
                   aria-label="Pin latitude" value={fields.lat} onChange={(e) => typeField("lat", e.target.value)}
                   data-testid="input-pin-lat" />
            <input className={`${input} flex-1 basis-28`} inputMode="decimal" placeholder="Longitude"
                   aria-label="Pin longitude" value={fields.lng} onChange={(e) => typeField("lng", e.target.value)}
                   data-testid="input-pin-lng" />
            <Button type="button" variant="outline" className="h-10" onClick={pinAtCentre}>
              <Crosshair /> Pin at map centre
            </Button>
          </div>
          <div className="flex gap-2">
            <Button type="button" variant="outline" className="flex-1" onClick={onClose}>Cancel</Button>
            <Button type="button" className="flex-1" disabled={!pin} onClick={() => pin && onUse(pin)}
                    data-testid="button-use-picked-place">
              Use this place
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
