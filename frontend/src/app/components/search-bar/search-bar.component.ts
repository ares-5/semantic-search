import { ChangeDetectionStrategy, Component, EventEmitter, inject, Output } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { LocaleService } from '../../core/services/locale.service';

@Component({
  selector: 'app-search-bar',
  imports: [FormsModule],
  templateUrl: './search-bar.component.html',
  styleUrl: './search-bar.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SearchBarComponent {
  private localeService: LocaleService = inject(LocaleService);

  query: string = '';
  @Output() search = new EventEmitter<string>();

  get placeholder(): string {
    return this.localeService.locale() === 'sr' ? 'Pretraži proizvode...' : 'Search products...';
  }

  get searchText(): string {
    return this.localeService.locale() === 'sr' ? 'Pretraži' : 'Search';
  }

  onSearch() {
    if (this.query.trim()) {
      this.search.emit(this.query);
    }
  }
}
