export interface Product {
  _id: string;
  title: string;
  description: string;
  details: string[];
  brand: string;
  sub_category: string;
  seller: string;
  embedding_en?: number[];
  embedding_sr?: number[];
}
