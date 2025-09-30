import { Injectable, signal, WritableSignal } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class LocaleService {
  private _locale: WritableSignal<'en' | 'sr'> = signal<'en' | 'sr'>(
    (localStorage.getItem('locale') as 'en' | 'sr') || 'en'
  );

  locale(): 'en' | 'sr' {
    return this._locale();
  }

  setLocale(locale: 'en' | 'sr') {
    this._locale.set(locale);
    localStorage.setItem('locale', locale);
  }
}
