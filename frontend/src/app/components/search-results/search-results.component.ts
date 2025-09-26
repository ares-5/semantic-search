import { ChangeDetectionStrategy, Component, Input } from '@angular/core';
import { PhDDissertation } from '../../core/models/phd-dissertation';

@Component({
  selector: 'app-search-results',
  imports: [],
  templateUrl: './search-results.component.html',
  styleUrl: './search-results.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class SearchResultsComponent {
  @Input() phdDissertations: PhDDissertation[] = [];
}
