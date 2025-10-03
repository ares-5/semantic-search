import { HttpClient } from '@angular/common/http';
import { inject, Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { SearchMode } from '../models/search-mode';
import { PhDDissertation } from '../models/phd-dissertation';

@Injectable({
  providedIn: 'root'
})
export class SearchService {
  private apiUrl: string = 'http://localhost:8000';
  private httpClient: HttpClient = inject(HttpClient);

  search(
    query: string,
    lang: 'en' | 'sr' = 'en',
    mode: SearchMode = SearchMode.SEMANTIC,
    size: number = 10,
    candidate_pool: number = 300,
    alpha: number = 0.65
  ): Observable<PhDDissertation[]> {
    return this.httpClient.get<PhDDissertation[]>(`${this.apiUrl}/search`, {
      params: {
        query: query,
        mode,
        lang,
        size: size.toString(),
        candidate_pool: candidate_pool,
        alpha: alpha
      }
    });
  }

  getById(id: string): Observable<PhDDissertation> {
    return this.httpClient.get<PhDDissertation>(`${this.apiUrl}/dissertations/${id}`);
  }
}
