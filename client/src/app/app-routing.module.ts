import { TeacherResponseComponent } from './teacher-response/teacher-response.component';
import { StudentSubmissionPageComponent } from './student-submission-page/student-submission-page.component';
import { StudentExamPageComponent } from './student-exam-page/student-exam-page.component';
import { StudentPhotoAudioComponent } from './student-photo-audio/student-photo-audio.component';
import { StudentInstructionComponent } from './student-instruction/student-instruction.component';
import { StudentDetailComponent } from './student-detail/student-detail.component';
import { DashboardComponent } from './dashboard/dashboard.component';
import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import { RegisterComponent } from './register/register.component';
import { LoginComponent } from './login/login.component';
import { AppComponent } from './app.component';
import { CreatetestComponent } from './createtest/createtest.component';
import {CreatetestDetailComponent} from './createtest-detail/createtest-detail.component';
import {StudentDashboardComponent} from './student-dashboard/student-dashboard.component';

const routes: Routes = [
  {
    path: "", component: AppComponent, children: [
      { path: "", component: RegisterComponent },
      { path: "login", component: LoginComponent },
      {path:"dashboard",component:DashboardComponent},
      {path:"createtest",component:CreatetestComponent},
      {path:"testdetail",component: CreatetestDetailComponent},
      {path:"studentdashboard",component:StudentDashboardComponent},
      {path:"studentdetail",component:StudentDetailComponent},
      {path:"studentinstruction",component:StudentInstructionComponent},
      {path:"photoaudio",component:StudentPhotoAudioComponent},
      {path:"exampage",component:StudentExamPageComponent},
      {path:"submission",component:StudentSubmissionPageComponent},
      {path:"response",component:TeacherResponseComponent}
    ]
  }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
