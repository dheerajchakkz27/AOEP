import { ComponentFixture, TestBed } from '@angular/core/testing';

import { CreatetestDetailComponent } from './createtest-detail.component';

describe('CreatetestDetailComponent', () => {
  let component: CreatetestDetailComponent;
  let fixture: ComponentFixture<CreatetestDetailComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ CreatetestDetailComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(CreatetestDetailComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
