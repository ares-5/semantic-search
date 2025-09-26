import { ChangeDetectionStrategy, Component, inject, signal, WritableSignal } from '@angular/core';
import { SearchService } from '../../core/services/search.service';
import { PhDDissertation } from '../../core/models/phd-dissertation';
import { SearchResultsComponent } from "../search-results/search-results.component";
import { SearchBarComponent } from "../search-bar/search-bar.component";

@Component({
  selector: 'app-catalog',
  imports: [SearchResultsComponent, SearchBarComponent],
  templateUrl: './catalog.component.html',
  styleUrls: ['./catalog.component.css'],
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class CatalogComponent {
  private searchService: SearchService = inject(SearchService);

  phdDissertations: WritableSignal<PhDDissertation[]> = signal([]);
  loading: boolean = false;

  onSearch(query: string) {
    this.loading = true;

    this.searchService.search(query).subscribe({
      next: (res: PhDDissertation[]) => {
        this.phdDissertations.set(res);
        this.loading = false;
      },
      error: (err) => {
        console.error(err);
        this.loading = false;
      }
    });
  }
 }
