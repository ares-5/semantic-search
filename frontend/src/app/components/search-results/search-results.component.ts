import { ChangeDetectionStrategy, Component, inject, signal, WritableSignal } from '@angular/core';
import { PhDDissertation } from '../../core/models/phd-dissertation';
import { SearchService } from '../../core/services/search.service';
import { LocaleService } from '../../core/services/locale.service';
import { SearchBarComponent } from '../search-bar/search-bar.component';
import { ActivatedRoute, Router } from '@angular/router';

@Component({
  selector: 'app-search-results',
  imports: [SearchBarComponent],
  templateUrl: './search-results.component.html',
  styleUrl: './search-results.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SearchResultsComponent {
  private searchService = inject(SearchService);
  private localeService = inject(LocaleService);
  private activatedRoute: ActivatedRoute = inject(ActivatedRoute);
  private router: Router = inject(Router);

  phdDissertations: WritableSignal<PhDDissertation[]> = signal([]);
  loading = false;

  get locale() {
    return this.localeService.locale();
  }

  ngOnInit() {
    this.activatedRoute.queryParams.subscribe(params => {
      const q = params['q'];
      if (q) {
        this.onSearch(q);
      }
    });
  }

  getLocalizedTitle(phd: PhDDissertation): string {
    const lang = this.locale as 'en' | 'sr';
    return phd.title[lang] ?? '';
  }

  getLocalizedDetails(phd: PhDDissertation): string {
    const lang = this.locale as 'en' | 'sr';
    return phd.details[lang] ?? '';
  }

  onSearch(query: string) {
    this.loading = true;
    this.router.navigate([], { queryParams: { q: query } });
    const lang = this.locale as 'en' | 'sr';

    this.searchService.search(query, lang).subscribe({
      next: (res) => {
        this.phdDissertations.set(res);
        this.loading = false;
      },
      error: (err) => {
        console.error(err);
        this.loading = false;
      }
    });
  }

  onClick(id: string) {
    this.router.navigate(['dissertation', id]);
  }
}
