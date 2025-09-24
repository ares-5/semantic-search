import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { SearchMode } from '../models/search-mode';
import { Product } from '../models/product';

@Injectable({
  providedIn: 'root'
})
export class SearchService {
  private apiUrl: string = 'http://localhost:8000';
  private httpClient: HttpClient = inject(HttpClient);

  search(
    query: string,
    mode: SearchMode = SearchMode.STANDARD,
    lang: string = 'en',
    size: number = 10,
    alpha: number = 0.5
  ): Observable<Product[]> {
    return this.httpClient.get<Product[]>(`${this.apiUrl}/search/`, {
      params: {
        query,
        mode,
        lang,
        size: size.toString(),
        alpha: alpha.toString()
      }
    });
  }
}
