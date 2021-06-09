import { NgModule } from '@angular/core';
import { BrowserModule } from '@angular/platform-browser';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { RegisterComponent } from './register/register.component';
import { LoginComponent } from './login/login.component';
import { HeaderComponent } from './header/header.component';
import { DashboardComponent } from './dashboard/dashboard.component';
import { CreatetestComponent } from './createtest/createtest.component';
import { CreatetestDetailComponent } from './createtest-detail/createtest-detail.component';
import { StudentDashboardComponent } from './student-dashboard/student-dashboard.component';
import { StudentDetailComponent } from './student-detail/student-detail.component';
import { StudentInstructionComponent } from './student-instruction/student-instruction.component';
import { StudentPhotoAudioComponent } from './student-photo-audio/student-photo-audio.component';
import { StudentExamPageComponent } from './student-exam-page/student-exam-page.component';
import { StudentSubmissionPageComponent } from './student-submission-page/student-submission-page.component';
import { TeacherResponseComponent } from './teacher-response/teacher-response.component';

@NgModule({
  declarations: [
    AppComponent,
    RegisterComponent,
    LoginComponent,
    HeaderComponent,
    DashboardComponent,
    CreatetestComponent,
    CreatetestDetailComponent,
    StudentDashboardComponent,
    StudentDetailComponent,
    StudentInstructionComponent,
    StudentPhotoAudioComponent,
    StudentExamPageComponent,
    StudentSubmissionPageComponent,
    TeacherResponseComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
