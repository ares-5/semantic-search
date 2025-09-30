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

  changeLocale(event: Event) {
    const select = event.target as HTMLSelectElement;
    this.localeService.setLocale(select.value as 'en' | 'sr');
  }

  goHome() {
    this.router.navigate(['/']);
  }
}
