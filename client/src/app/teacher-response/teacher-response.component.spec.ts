import { ComponentFixture, TestBed } from '@angular/core/testing';

import { TeacherResponseComponent } from './teacher-response.component';

describe('TeacherResponseComponent', () => {
  let component: TeacherResponseComponent;
  let fixture: ComponentFixture<TeacherResponseComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ TeacherResponseComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(TeacherResponseComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
