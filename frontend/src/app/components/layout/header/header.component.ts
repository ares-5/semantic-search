import { Component, effect, inject, Signal, signal, WritableSignal } from '@angular/core';
import { LocaleService } from '../../../core/services/locale.service';
import { Router } from '@angular/router';
import { SearchMode } from '../../../core/models/search-mode';

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.css']
})
export class HeaderComponent {
  public localeService: LocaleService = inject(LocaleService);
  private router: Router = inject(Router);

  private readonly _searchOptions: WritableSignal<Record<SearchMode, string> | undefined> = signal<Record<SearchMode, string> | undefined>(undefined);
  readonly searchOptions: Signal<Record<SearchMode, string> | undefined> = this._searchOptions.asReadonly();

  readonly searchOptionValues: SearchMode[] = Object.values(SearchMode);

  get locale() {
    return this.localeService.locale();
  }

  get searchMode() {
    return this.localeService.searchMode();
  }

  constructor() {
    effect(() => {
      this.localeService.locale();
      const options: Record<SearchMode, string> = {} as Record<SearchMode, string>;

      for (const mode of Object.values(SearchMode)) {
        options[mode] = this.getLocalizedSearchMode(mode);
      }

      this._searchOptions.set(options);
    });
  }

  getLocalizedAppName(): string {
    const lang: "en" | "sr" = this.locale as 'en' | 'sr';
    return lang === 'en' ? 'PhD dissertations' : 'Doktorske disertacije';
  }

  getLocalizedSearchMode(mode: SearchMode): string {
    switch (mode) {
      case SearchMode.STANDARD:
        return this.locale === 'en' ? 'Standard search' : 'Standardna pretraga';
      case SearchMode.SEMANTIC:
        return this.locale === 'en' ? 'Semantic search' : 'Semantička pretraga';
      case SearchMode.HYBRID:
        return this.locale === 'en' ? 'Hybrid search' : 'Hibridna pretraga';
      case SearchMode.RERANKED:
        return this.locale === 'en' ? 'Reranked search' : 'Rangirajuća pretraga'
      default:
        throw new Error();
    }
  }

  changeSearchMode(event: Event): void {
    const select = event.target as HTMLSelectElement;
    this.localeService.setSearchMode(select.value as SearchMode);
  }

  changeLocale(event: Event): void {
    const select: HTMLSelectElement = event.target as HTMLSelectElement;
    this.localeService.setLocale(select.value as 'en' | 'sr');
  }

  goHome() {
    this.router.navigate(['/']);
  }
}
