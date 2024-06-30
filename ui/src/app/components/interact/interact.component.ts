import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatIconModule } from '@angular/material/icon';
import { MatMenuModule } from '@angular/material/menu';
import { MatButtonModule } from '@angular/material/button';

interface Message {
  text: string;
  user: 'bot' | 'user';
  audioUrl: boolean;
}

@Component({
  selector: 'app-interact',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatIconModule,
    MatMenuModule,
    MatButtonModule,
  ],
  templateUrl: './interact.component.html',
  styleUrl: './interact.component.scss',
})
export class InteractComponent {
  messages: Message[] = [
    { text: 'Hello! How can I assist you today?', user: 'bot', audioUrl: true },
  ];
  newMessage: string = '';

  sendMessage(): void {
    if (this.newMessage.trim()) {
      this.messages.push({
        text: this.newMessage,
        user: 'user',
        audioUrl: true,
      });
      this.newMessage = '';
      this.addBotResponse();
    }
  }

  addBotResponse(): void {
    setTimeout(() => {
      this.messages.push({
        text: 'This is a bot response.',
        user: 'bot',
        audioUrl: true,
      });
    }, 1000);
  }
}
