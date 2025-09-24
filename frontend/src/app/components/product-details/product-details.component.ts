import { ChangeDetectionStrategy, Component } from '@angular/core';

@Component({
  selector: 'app-product-details',
  imports: [],
  template: './product-details.component.html',
  styleUrl: './product-details.component.css',
  changeDetection: ChangeDetectionStrategy.OnPush,
})
export class ProductDetailsComponent { }
