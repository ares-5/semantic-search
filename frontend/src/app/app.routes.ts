import { Routes } from '@angular/router';
import { LayoutComponent } from './components/layout/layout.component';
import { SearchResultsComponent } from './components/search-results/search-results.component';
import { PhdDissertationComponent } from './components/phd-dissertation/phd-dissertation.component';

export const routes: Routes = [
    {
        path: '',
        component: LayoutComponent,
        children: [
            {
                path: '',
                component: SearchResultsComponent,
            },
            {
                path: 'dissertation/:id',
                component: PhdDissertationComponent
            }
        ]
    }
];
