import axios from "axios";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function health() {
  const { data } = await axios.get(`${API}/health`);
  return data;
}

export async function detect(file) {
  const fd = new FormData();
  fd.append("image", file);
  const { data } = await axios.post(`${API}/detect`, fd);
  return data;
}

export async function protect(file, epsilon, useFacenet) {
  const fd = new FormData();
  fd.append("image", file);
  if (epsilon != null) fd.append("epsilon", epsilon);
  if (useFacenet) fd.append("use_facenet", "true");
  const { data } = await axios.post(`${API}/protect`, fd);
  return data;
}
