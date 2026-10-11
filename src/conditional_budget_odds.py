"""Read-only, just-in-time odds for conditional budgets; no shared caches."""
import re
from urllib.parse import urlsplit
from conditional_budget import number


def _get_html(url):
    import requests
    response = requests.get(url, timeout=(3, 6), headers={
        'User-Agent': 'Mozilla/5.0', 'Cache-Control': 'no-cache'})
    response.raise_for_status()
    response.encoding = 'utf-8'
    return response.text


def _winticket(html, row):
    from fetch_today_entries import (extract_preloaded_state, find_query_data,
                                    normalize_date, get_venue_name, build_odds_rows)
    state = extract_preloaded_state(html)
    data = find_query_data(state, 'FETCH_KEIRIN_RACE')
    race = data['race']
    if (str(race['id']) != str(row['race_id'])
            or normalize_date(data['schedule']['date']) != row['date']
            or int(race['number']) != int(float(row['race_no']))
            or get_venue_name(state, data) != row['venue']):
        raise ValueError('winticket race identity mismatch')
    absent = {str(e.get('number')) for e in data.get('entries', []) if e.get('absent')}
    if any(set(t.split('-')) & absent for t in row['top12']):
        raise ValueError('withdrawn rider in selected tickets')
    raw = build_odds_rows(find_query_data(state, 'FETCH_KEIRIN_RACE_ODDS'),
                          row['date'], row['venue'], int(float(row['race_no'])),
                          str(row['race_id']), row['source_url'])
    quotes = {}
    for item in raw:
        if item['bet_type'] != 'trifecta':
            continue
        if item['buy'] in quotes:
            raise ValueError('duplicate quote')
        quotes[item['buy']] = item['odds_used']
    return quotes


def _oddspark(html, row):
    from market_odds_sources import parse_oddspark
    return parse_oddspark(html, row['date'].replace('-', ''),
                          int(float(row['race_no'])), row['venue'])


def capture(row, clock, get_html=None):
    rid = str(row['race_id'])
    if (not re.fullmatch(r'\d{12}', rid)
            or rid[4:] != row['date'].replace('-', '')
            or int(rid[:2]) != int(float(row['race_no']))):
        raise ValueError('invalid race identity')
    url = str(row.get('source_url', ''))
    parsed = urlsplit(url)
    match = re.fullmatch(r'/keirin/([^/]+)/racecard/(\d{10})/(\d+)/(\d+)/?', parsed.path)
    if (parsed.scheme != 'https' or parsed.hostname != 'www.winticket.jp'
            or match is None or match[2][-2:] != rid[2:4]
            or int(match[4]) != int(rid[:2])):
        raise ValueError('invalid prediction source URL')
    get_html = get_html or _get_html
    alternatives = [
        ('winticket', url, _winticket),
        ('oddspark', 'https://www.oddspark.com/keirin/Odds.do?betType=9&joCode='
         + rid[2:4] + '&kaisaiBi=' + rid[4:] + '&raceNo=' + str(int(rid[:2]))
         + '&viewType=1', _oddspark)]
    failures = []
    for provider, source, parser in alternatives:
        now = clock()
        if (now.tzinfo is None or now.date().isoformat() != row['date']
                or now.timestamp() >= float(row['close_at'])):
            raise ValueError('quote request outside pre-close current-day window')
        try:
            quotes = parser(get_html(source), row)
            selected = {t: number(quotes[t]) for t in row['top12']}
            if any(p >= number('9999.9') for p in selected.values()):
                raise ValueError('censored odds cannot establish exact payout')
            captured = clock()
            if captured.timestamp() >= float(row['close_at']):
                raise ValueError('closed during odds capture')
            return {key: row[key] for key in ('race_id', 'date', 'venue', 'race_no')} | {
                'bet_type': 'trifecta', 'provider': provider, 'source_url': source,
                'captured_at_jst': captured.isoformat(),
                'odds': {t: str(v) for t, v in selected.items()}}
        except Exception as error:
            failures.append(provider + ': ' + str(error)[:100])
            if 'withdrawn rider' in str(error):
                break
    raise ValueError('; '.join(failures))
