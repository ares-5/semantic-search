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

  get locale() {
    return this.localeService.locale();
  }

  get placeholder(): string {
    const lang = this.locale as 'en' | 'sr';
    return lang === 'sr' ? 'Pretraži proizvode...' : 'Search products...';
  }

  get searchText(): string {
    const lang = this.locale as 'en' | 'sr';
    return lang === 'sr' ? 'Pretraži' : 'Search';
  }

  onSearch() {
    if (this.query.trim()) {
      this.search.emit(this.query);
    }
  }
}
