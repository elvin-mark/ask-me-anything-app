import { Component } from '@angular/core';
import { FileUploaderComponent } from '../file-uploader/file-uploader.component';
import { MatMenuModule } from '@angular/material/menu';
import { MatButtonModule } from '@angular/material/button';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-sources',
  standalone: true,
  imports: [
    FileUploaderComponent,
    MatMenuModule,
    MatButtonModule,
    CommonModule,
  ],
  templateUrl: './sources.component.html',
  styleUrl: './sources.component.scss',
})
export class SourcesComponent {
  isNewFile: boolean = false;
  isNewRawText: boolean = false;
  isNewURL: boolean = false;

  openFile() {
    this.isNewFile = true;
    this.isNewRawText = false;
    this.isNewURL = false;
  }

  openRawText() {
    this.isNewFile = false;
    this.isNewRawText = true;
    this.isNewURL = false;
  }

  openURL() {
    this.isNewFile = false;
    this.isNewRawText = false;
    this.isNewURL = true;
  }
}
