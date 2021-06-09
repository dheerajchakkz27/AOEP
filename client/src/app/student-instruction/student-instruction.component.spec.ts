import { ComponentFixture, TestBed } from '@angular/core/testing';

import { StudentInstructionComponent } from './student-instruction.component';

describe('StudentInstructionComponent', () => {
  let component: StudentInstructionComponent;
  let fixture: ComponentFixture<StudentInstructionComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ StudentInstructionComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(StudentInstructionComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
