import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NLP Pipeline — Multi-Task Analyzer",
  description:
    "Real-time multi-task NLP: joint sentiment, emotion, and toxicity from a single RoBERTa forward pass, with named-entity extraction.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
