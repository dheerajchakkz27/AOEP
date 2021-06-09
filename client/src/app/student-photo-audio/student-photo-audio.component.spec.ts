import { ComponentFixture, TestBed } from '@angular/core/testing';

import { StudentPhotoAudioComponent } from './student-photo-audio.component';

describe('StudentPhotoAudioComponent', () => {
  let component: StudentPhotoAudioComponent;
  let fixture: ComponentFixture<StudentPhotoAudioComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ StudentPhotoAudioComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(StudentPhotoAudioComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
