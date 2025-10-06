import { Component, inject } from '@angular/core';
import { LocaleService } from '../../../core/services/locale.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-header',
  templateUrl: './header.component.html',
  styleUrls: ['./header.component.css']
})
export class HeaderComponent {
  public localeService: LocaleService = inject(LocaleService);
  private router: Router = inject(Router);

  getLocalizedAppName(): string {
    const lang = this.locale as 'en' | 'sr';
    return lang === 'en' ? 'PhD dissertations' : 'Doktorske disertacije';
  }

  changeLocale(event: Event): void {
    const select = event.target as HTMLSelectElement;
    this.localeService.setLocale(select.value as 'en' | 'sr');
  }

  get locale() {
    return this.localeService.locale();
  }

  goHome() {
    this.router.navigate(['/']);
  }
}
