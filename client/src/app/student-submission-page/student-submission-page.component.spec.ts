import { ComponentFixture, TestBed } from '@angular/core/testing';

import { StudentSubmissionPageComponent } from './student-submission-page.component';

describe('StudentSubmissionPageComponent', () => {
  let component: StudentSubmissionPageComponent;
  let fixture: ComponentFixture<StudentSubmissionPageComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ StudentSubmissionPageComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(StudentSubmissionPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
