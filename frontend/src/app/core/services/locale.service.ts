import { Injectable, Signal, signal, WritableSignal } from '@angular/core';
import { SearchMode } from '../models/search-mode';

@Injectable({ providedIn: 'root' })
export class LocaleService {
  private readonly _locale: WritableSignal<'en' | 'sr'> = signal<'en' | 'sr'>(
    (localStorage.getItem('locale') as 'en' | 'sr') || 'en'
  );
  readonly locale: Signal<'en' | 'sr'> = this._locale.asReadonly();

  private readonly _searchMode: WritableSignal<SearchMode> = signal<SearchMode>(SearchMode.STANDARD);
  readonly searchMode: Signal<SearchMode> = this._searchMode.asReadonly();

  getSearchMode(): string {
    switch (this.searchMode()) {
      case SearchMode.STANDARD:
        return this.locale() === 'en' ? 'Standard search' : 'Standardna pretraga';
      case SearchMode.SEMANTIC:
        return this.locale() === 'en' ? 'Semantic search' : 'Semantička pretraga';
      case SearchMode.HYBRID:
        return this.locale() === 'en' ? 'Hybrid search' : 'Hibridna pretraga';
      case SearchMode.RERANKED:
        return this.locale() === 'en' ? 'Reranked search' : 'Rangirajuća pretraga'
    }
  }

  setLocale(locale: 'en' | 'sr') {
    this._locale.set(locale);
    localStorage.setItem('locale', locale);
  }

  setSearchMode(searchMode: SearchMode): void {
    this._searchMode.set(searchMode);
  }
}
