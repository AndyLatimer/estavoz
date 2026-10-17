// Datos públicos de la base de datos de respuestas (Supabase). La clave "anon" es pública
// por diseño: la base de datos solo le deja añadir respuestas (despliegue/supabase.sql).
// Si están vacíos, la pregunta "¿Acertamos?" no se muestra.
window.ESTAVOZ_CONFIG = {
  supabaseUrl: "",
  supabaseKey: "",
};
