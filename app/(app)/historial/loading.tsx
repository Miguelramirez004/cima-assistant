import { Esqueleto } from "@/components/motion";

export default function HistorialLoading() {
  return (
    <div role="status" aria-label="Cargando historial">
      <Esqueleto lineas={2} className="mb-8 max-w-md" />
      <ul className="divide-y divide-line rounded-xl border border-line bg-white">
        {[0, 1, 2, 3].map((i) => (
          <li key={i} className="px-5 py-4"><Esqueleto lineas={2} className="max-w-lg" /></li>
        ))}
      </ul>
    </div>
  );
}
