import { DashboardComponent } from './dashboard/dashboard.component';
import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { RegisterComponent } from './register/register.component';
import { LoginComponent } from './login/login.component';
import { AppComponent } from './app.component';
import { CreatetestComponent } from './createtest/createtest.component';

const routes: Routes = [
  {
    path: "", component: AppComponent, children: [
      { path: "", component: RegisterComponent },
      { path: "login", component: LoginComponent },
      {path:"dashboard",component:DashboardComponent},
      {path:"createtest",component:CreatetestComponent}
    ]
  }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
