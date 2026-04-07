function highlightMatch(suggestion, query) {
  const cleanQuery = query.trim().toLowerCase();
  if (!cleanQuery) {
    return suggestion;
  }

  const idx = suggestion.toLowerCase().indexOf(cleanQuery);
  if (idx === -1) {
    return suggestion;
  }

  const before = suggestion.slice(0, idx);
  const match = suggestion.slice(idx, idx + cleanQuery.length);
  const after = suggestion.slice(idx + cleanQuery.length);

  return (
    <>
      {before}
      <strong>{match}</strong>
      {after}
    </>
  );
}

export default function SuggestionBox({ suggestions, query, loading, error }) {
  if (!query.trim()) {
    return null;
  }

  if (loading) {
    return <div className="suggestions loading">Loading suggestions...</div>;
  }

  if (error) {
    return <div className="suggestions error">{error}</div>;
  }

  if (!suggestions.length) {
    return <div className="suggestions empty">No suggestions found.</div>;
  }

  return (
    <ul className="suggestions" role="listbox" aria-label="Keyword suggestions">
      {suggestions.map((item) => (
        <li key={item} className="suggestion-item">
          {highlightMatch(item, query)}
        </li>
      ))}
    </ul>
  );
}
