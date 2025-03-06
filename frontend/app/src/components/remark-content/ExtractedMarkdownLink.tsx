import React from 'react';

interface ExtractedMarkdownLinkProps {
  markdown: string;
}

export function ExtractedMarkdownLink({ markdown }: ExtractedMarkdownLinkProps) {
  // Regular expression to match markdown image syntax: ![alt](url)
  const regex = /!\[.*?\]\((.*?)\)/;
  const match = markdown.match(regex);
  const link = match ? match[1] : null;

  return (
    <div>
      {link ? (
        <>
          Extracted image link:{' '}
          <a href={link} target="_blank" rel="noopener noreferrer">
            {link}
          </a>
        </>
      ) : (
        'No image link found'
      )}
    </div>
  );
}
