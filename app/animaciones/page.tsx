import { notFound } from "next/navigation";

import { MotionShowcase } from "@/components/MotionShowcase";

// Catálogo de animaciones para trabajar la interfaz; no se publica en producción.
export default function AnimacionesPage() {
  if (process.env.NODE_ENV === "production") notFound();
  return <MotionShowcase />;
}
