import { Routes } from '@angular/router';
import { CatalogComponent } from './components/catalog/catalog.component';
import { LayoutComponent } from './components/layout/layout.component';

export const routes: Routes = [
    {
        path: '',
        component: LayoutComponent,
        children: [
            {
                path: '',
                component: CatalogComponent
            }
        ]
    }
];
