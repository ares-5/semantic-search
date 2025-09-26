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
    mode: SearchMode = SearchMode.SEMANTIC,
    lang: string = 'sr',
    size: number = 10,
    alpha: number = 0.5
  ): Observable<PhDDissertation[]> {
    const encodedQuery = encodeURIComponent(query);

    return this.httpClient.get<PhDDissertation[]>(`${this.apiUrl}/search`, {
      params: {
        query: encodedQuery,
        mode,
        lang,
        size: size.toString(),
        alpha: alpha.toString()
      }
    });
  }
}
