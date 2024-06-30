import { Routes } from '@angular/router';
import { SourcesComponent } from './components/sources/sources.component';
import { InteractComponent } from './components/interact/interact.component';

export const routes: Routes = [
  {
    path: 'sources',
    component: SourcesComponent,
  },
  {
    path: 'interact',
    component: InteractComponent,
  },
];
