import { ChangeDetectionStrategy, Component, Input } from '@angular/core';
import { Product } from '../../core/models/product';

@Component({
  selector: 'app-search-results',
  imports: [],
  template: './search-results.component.html',
  styleUrl: './search-results.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SearchResultsComponent { 
  @Input() products: Product[] = [];
}
