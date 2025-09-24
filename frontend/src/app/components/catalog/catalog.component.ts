import { ChangeDetectionStrategy, Component, inject, signal, WritableSignal } from '@angular/core';
import { Product } from '../../core/models/product';
import { SearchService } from '../../core/services/search.service';
import { SearchResultsComponent } from '../search-results/search-results.component';

@Component({
  selector: 'app-catalog',
  imports: [SearchResultsComponent],
  template: './catalog.component.html',
  styleUrl: './catalog.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class CatalogComponent {
  private searchService: SearchService = inject(SearchService);

  products: WritableSignal<Product[]> = signal([]);
  loading: boolean = false;

  onSearch(query: string) {
    this.loading = true;

    this.searchService.search(query).subscribe({
      next: (res: Product[]) => {
        this.products.set(res);
        this.loading = false;
      },
      error: (err) => {
        console.error(err);
        this.loading = false;
      }
    });
  }
 }
