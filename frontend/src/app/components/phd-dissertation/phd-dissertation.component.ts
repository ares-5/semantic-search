import { Component, inject, OnInit } from '@angular/core';
import { PhDDissertation } from '../../core/models/phd-dissertation';
import { ActivatedRoute } from '@angular/router';
import { SearchService } from '../../core/services/search.service';
import { LocaleService } from '../../core/services/locale.service';

@Component({
  selector: 'app-phd-dissertation',
  templateUrl: './phd-dissertation.component.html',
  styleUrl: './phd-dissertation.component.css'
})
export class PhdDissertationComponent implements OnInit {
  private route: ActivatedRoute = inject(ActivatedRoute);
  private searchService: SearchService = inject(SearchService);
  private localeService: LocaleService = inject(LocaleService);

  dissertation?: PhDDissertation;
  loading = false;

  ngOnInit() {
    this.loading = true;
    const id = this.route.snapshot.paramMap.get('id');
    if (id) {
      this.searchService.getById(id).subscribe({
        next: (res) => {
          this.dissertation = res;
          this.loading = false;
        },
        error: (err) => {
          console.error(err);
          this.loading = false;
        }
      });
    }
  }

  get locale() {
    return this.localeService.locale();
  }

  getLocalizedTitle(): string {
    if (!this.dissertation) return '';
    const lang = this.locale as 'en' | 'sr';
    return this.dissertation.title[lang] ?? '';
  }

  getLocalizedDetails(): string {
    if (!this.dissertation) return '';
    const lang = this.locale as 'en' | 'sr';
    return this.dissertation.details[lang] ?? '';
  }

  goBack() {
    window.history.back();
  }
}
