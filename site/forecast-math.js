/* Exact trifecta formations. No odds lookup and no selection changes. */
(function (root, factory) {
  'use strict';
  var api = factory();
  if (typeof module === 'object' && module.exports) module.exports = api;
  if (root) root.ForecastMath = api;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  'use strict';

  function ticket(value) {
    if (typeof value !== 'string' || !/^[1-9]-[1-9]-[1-9]$/.test(value)) {
      throw new TypeError('A trifecta ticket must have the form 1-2-3.');
    }
    var parts = value.split('-').map(Number);
    if (new Set(parts).size !== 3) throw new TypeError('A trifecta needs three different racers.');
    return parts;
  }

  function axis(values) {
    if (!Array.isArray(values)) throw new TypeError('Formation positions must be arrays.');
    return Array.from(new Set(values.map(function (value) {
      var n = Number(value);
      if (!Number.isInteger(n) || n < 1 || n > 9) throw new TypeError('Invalid racer number.');
      return n;
    }))).sort(function (a, b) { return a - b; });
  }

  function expandFormation(formation) {
    var first = axis(formation.first);
    var second = axis(formation.second);
    var third = axis(formation.third);
    var buys = [];
    first.forEach(function (a) {
      second.forEach(function (b) {
        if (a === b) return;
        third.forEach(function (c) {
          if (a !== c && b !== c) buys.push([a, b, c].join('-'));
        });
      });
    });
    return buys;
  }

  function fromAxes(axes) {
    var formation = { first: axes[0], second: axes[1], third: axes[2] };
    formation.tickets = expandFormation(formation);
    formation.count = formation.tickets.length;
    return formation;
  }

  function width(formation) {
    return formation.first.length + formation.second.length + formation.third.length;
  }

  function better(candidate, incumbent) {
    return candidate.count > incumbent.count
      || (candidate.count === incumbent.count && width(candidate) < width(incumbent));
  }

  function grow(seed, remaining, parts) {
    var current = fromAxes(parts.get(seed).map(function (n) { return [n]; }));
    while (true) {
      var best = current;
      remaining.forEach(function (buy) {
        if (current.tickets.indexOf(buy) !== -1) return;
        var p = parts.get(buy);
        // Add a whole ticket at once: mirrored positions may need two axes
        // to grow together before they generate any additional valid ticket.
        var candidate = fromAxes([current.first, current.second, current.third].map(function (a, i) {
          return axis(a.concat(p[i]));
        }));
        if (candidate.count > current.count && better(candidate, best)
            && candidate.tickets.every(function (t) { return remaining.has(t); })) best = candidate;
      });
      if (best === current) return current;
      current = best;
    }
  }

  function toFormations(tickets) {
    if (!Array.isArray(tickets)) throw new TypeError('Tickets must be an array.');
    var parts = new Map();
    tickets.forEach(function (buy) { parts.set(buy, ticket(buy)); });
    var remaining = new Set(Array.from(parts.keys()).sort());
    var result = [];
    while (remaining.size) {
      var best = null;
      remaining.forEach(function (seed) {
        var candidate = grow(seed, remaining, parts);
        if (!best || better(candidate, best)) best = candidate;
      });
      result.push(best);
      // Every output row is a subset of what is still uncovered, so rows
      // neither add unselected tickets nor ask the user to buy a ticket twice.
      best.tickets.forEach(function (buy) { remaining.delete(buy); });
    }
    return result;
  }

  function stakeFormations(allocations) {
    if (!Array.isArray(allocations)) throw new TypeError('Allocations must be an array.');
    var groups = new Map();
    var seen = new Set();
    allocations.forEach(function (entry) {
      ticket(entry.buy);
      if (seen.has(entry.buy)) throw new TypeError('Duplicate allocation ticket.');
      seen.add(entry.buy);
      if (!Number.isSafeInteger(entry.stake_yen) || entry.stake_yen < 100 || entry.stake_yen % 100) {
        throw new TypeError('Each stake must be a positive multiple of 100 yen.');
      }
      if (!groups.has(entry.stake_yen)) groups.set(entry.stake_yen, []);
      groups.get(entry.stake_yen).push(entry.buy);
    });
    var result = [];
    Array.from(groups.keys()).sort(function (a, b) { return b - a; }).forEach(function (amount) {
      toFormations(groups.get(amount)).forEach(function (formation) {
        formation.stake_yen = amount;
        formation.total_stake_yen = amount * formation.count;
        result.push(formation);
      });
    });
    return result;
  }

  return Object.freeze({
    toFormations: toFormations,
    expandFormation: expandFormation,
    stakeFormations: stakeFormations
  });
});
