import { MANAGE_PERMISSIONS, PERMISSION_LINE, PHOTO_PERMISSIONS_URL } from "@/lib/publicView";

/** One of the two approved lines to photographers, with its link (lib/publicView.ts). */
export function PhotoPermissionLine({ kind }: { kind: keyof typeof PERMISSION_LINE }) {
  return (
    <>
      {PERMISSION_LINE[kind]}{" "}
      <a href={PHOTO_PERMISSIONS_URL} target="_blank" rel="noreferrer" className="underline"
         data-testid="link-photo-permissions">
        {MANAGE_PERMISSIONS}
      </a>
    </>
  );
}
