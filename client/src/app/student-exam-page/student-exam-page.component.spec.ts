import { ComponentFixture, TestBed } from '@angular/core/testing';

import { StudentExamPageComponent } from './student-exam-page.component';

describe('StudentExamPageComponent', () => {
  let component: StudentExamPageComponent;
  let fixture: ComponentFixture<StudentExamPageComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ StudentExamPageComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(StudentExamPageComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
