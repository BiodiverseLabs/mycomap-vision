import { Link, useLocation } from "wouter";
import { useEffect, useState } from "react";
import { ArrowUpRight, Menu, X } from "lucide-react";
import { api, signInUrl, type Me } from "@/lib/api";

// Header and footer follow mycomap.org's PublicLayout and MainNavigation so the
// two sites read as one: sticky white header, logo + "MycoMap" in brown, grey
// links that turn myco-green, brown gradient footer.

const NAV = [
  { href: "/", label: "Identify" },
  { href: "/models", label: "Models" },
  { href: "/data", label: "Data" },
  { href: "/about", label: "How it works" },
];

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative min-h-screen flex flex-col bg-white">
      <Header />
      <main className="flex-1 relative bg-white">{children}</main>
      <Footer />
    </div>
  );
}

function Header() {
  const [location] = useLocation();
  const [open, setOpen] = useState(false);
  const isActive = (href: string) => (href === "/" ? location === "/" : location.startsWith(href));
  const cls = (href: string) =>
    `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
      isActive(href)
        ? "bg-myco-green/10 text-myco-green"
        : "text-gray-700 hover:text-myco-green hover:bg-myco-green/5"
    }`;
  return (
    <header className="sticky top-0 z-50 w-full border-b border-myco-brown/10 bg-white/95 backdrop-blur supports-[backdrop-filter]:bg-white/80">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 items-center justify-between">
          <div className="flex items-center gap-8">
            <Link href="/" className="flex items-center gap-3">
              <img src="/mycomap-logo.png" alt="MycoMap Logo" className="h-10 w-auto" />
              <span className="hidden sm:flex items-baseline gap-2">
                <span className="text-xl font-semibold text-myco-brown">MycoMap</span>
                <span className="text-xl font-semibold text-myco-green">Vision</span>
              </span>
            </Link>
            <nav className="hidden lg:flex items-center gap-1">
              {NAV.map((n) => (
                <Link key={n.href} href={n.href} className={cls(n.href)}>
                  {n.label}
                </Link>
              ))}
            </nav>
          </div>
          <div className="flex items-center gap-2">
            <Account />
            <a
              href="https://mycomap.org"
              className="hidden sm:inline-flex items-center gap-1 px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:text-myco-green hover:bg-myco-green/5"
            >
              mycomap.org <ArrowUpRight className="h-3.5 w-3.5" />
            </a>
            <button
              className="lg:hidden p-2 rounded-md text-gray-700 hover:bg-myco-green/5"
              onClick={() => setOpen(!open)}
              aria-label="Menu"
            >
              {open ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
            </button>
          </div>
        </div>
        {open && (
          <nav className="lg:hidden pb-3 flex flex-col gap-1" onClick={() => setOpen(false)}>
            {NAV.map((n) => (
              <Link key={n.href} href={n.href} className={cls(n.href)}>
                {n.label}
              </Link>
            ))}
            <a href="https://mycomap.org" className={cls("#")}>
              mycomap.org
            </a>
          </nav>
        )}
      </div>
    </header>
  );
}

/** Who is signed in (with their mycomap.org account), when this site uses sign-in. */
function Account() {
  const [me, setMe] = useState<Me | null>(null);
  useEffect(() => {
    api.me().then(setMe).catch(() => setMe(null));
  }, []);
  if (!me || me.signin === "off") return null;
  const link = "px-3 py-2 rounded-md text-sm font-medium text-gray-700 hover:text-myco-green hover:bg-myco-green/5";
  if (!me.user) {
    return <a href={signInUrl()} className={link}>Sign in</a>;
  }
  return (
    <span className="flex items-center gap-1 text-sm text-gray-600">
      <span className="hidden md:inline">{me.user.name ?? "Signed in"}</span>
      <button
        className={link}
        onClick={async () => {
          location.href = await api.signOut();
        }}
      >
        Sign out
      </button>
    </span>
  );
}

function Footer() {
  return (
    <footer className="relative overflow-hidden mt-16">
      <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-transparent via-myco-green to-transparent" />
      <div className="absolute inset-0 bg-gradient-to-b from-[#8a5c3a] to-myco-brown" />
      <div className="relative z-10 container mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="flex flex-col md:flex-row justify-between gap-6">
          <div className="max-w-md">
            <div className="flex items-center gap-3 mb-3">
              <img src="/mycomap-logo.png" alt="" className="h-9 w-auto brightness-0 invert" />
              <span className="text-lg font-bold text-white">MycoMap Vision</span>
            </div>
            <p className="text-white/70 text-sm leading-relaxed">
              Photo identification trained only on DNA-verified MycoMap records, using every
              photo of a find. Early development: results are provisional.
            </p>
          </div>
          <ul className="space-y-2 text-sm">
            {[
              ["https://mycomap.org", "MycoMap.org"],
              ["https://mycomap.org/network", "Free sequencing"],
              ["https://mycomap.org/protocols", "Participation protocols"],
            ].map(([href, label]) => (
              <li key={href}>
                <a href={href} className="text-white/70 hover:text-myco-green transition-colors">
                  {label}
                </a>
              </li>
            ))}
          </ul>
        </div>
        <div className="border-t border-white/10 mt-8 pt-4 text-white/50 text-sm">
          &copy; {new Date().getFullYear()} MycoMap.org. Photos belong to their photographers.
        </div>
      </div>
    </footer>
  );
}

export function PageHeader({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="bg-[#f8f5f0] border-b border-[#A87146]/10">
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <h1 className="font-display text-3xl md:text-4xl text-[#4a3728]">{title}</h1>
        {children && <div className="mt-2 text-[#5c4a3a] max-w-3xl">{children}</div>}
      </div>
    </div>
  );
}
