# Atlas API Reference (v2)

Version 2.8 | Effective 1 March 2025 | Owner: Ivan Petrusek, Staff Engineer, Platform

## Overview

The Atlas API serves Northwind Cartography map, geocoding and routing data over HTTPS. The base URL is https://api.northwindcarto.example/v2 and every response is JSON encoded as UTF-8. Requests to the v1 base URL are refused; v1 was retired on 30 September 2024.

## Authentication

Every request carries an OAuth 2.0 bearer token in the Authorization header. Access tokens expire after 12 hours and refresh tokens expire after 30 days of inactivity. Tokens are scoped: maps.read, maps.write, geocode, routes and usage.read. A request with a valid token but the wrong scope is refused with 403 insufficient_scope, which is the error most often mistaken for an expired token.

## Rate limits

Rate limits are enforced per API key on a sliding one-minute window. The Free tier allows 60 requests per minute, the Standard tier 600 requests per minute, and the Enterprise tier 4,000 requests per minute. Every tier tolerates a burst of twice its limit for up to 10 seconds before shedding load.

A throttled request returns 429 rate_limited with a Retry-After header giving the number of seconds to wait. Clients should honour Retry-After rather than retrying immediately; a sustained throttling rate is treated as a production incident under the thresholds in the Incident Response Runbook.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| GET | /maps | List map definitions for the calling account |
| POST | /maps | Create a map definition from a GeoJSON payload |
| GET | /maps/{map_id} | Fetch one map definition |
| GET | /maps/{map_id}/tiles/{z}/{x}/{y} | Fetch a rendered vector tile |
| POST | /geocode | Resolve up to 100 addresses in one call |
| POST | /routes | Compute a route between 2 and 25 waypoints |
| GET | /usage | Return the calling account's usage for a billing period |

## Payload limits

A GeoJSON upload to POST /maps may not exceed 25 megabytes or 10,000 features, whichever limit is reached first. A payload that parses but describes invalid geometry is rejected with 422 unprocessable_geometry and a list of the offending feature indexes. Rendered vector tiles are cached at the edge for 3,600 seconds and carry an ETag; a conditional request that matches returns 304.

## Pagination

List endpoints use cursor pagination. The page_size parameter defaults to 50 and may not exceed 200. A response containing more data returns a next_cursor value, which the client passes back as the cursor parameter. Cursors are opaque and expire after 15 minutes.

## Idempotency

POST /maps, POST /geocode and POST /routes accept an Idempotency-Key header. The key is a client-generated string of at most 128 characters, and Atlas retains the result of a keyed request for 24 hours. Replaying a key with a different request body returns 409 conflict.

## Error codes

| Status | Code | Meaning |
| --- | --- | --- |
| 400 | invalid_request | The request body or query string is malformed |
| 401 | invalid_token | The bearer token is missing, expired or revoked |
| 403 | insufficient_scope | The token is valid but lacks the required scope |
| 404 | not_found | No resource matches the identifier |
| 409 | conflict | Idempotency key reused with a different body |
| 422 | unprocessable_geometry | The geometry could not be interpreted |
| 429 | rate_limited | The account exceeded its per-minute rate limit |
| 500 | internal_error | Unexpected server fault; safe to retry with backoff |

## Webhooks

Atlas delivers asynchronous job results by webhook. Each delivery is signed with HMAC-SHA256 and the signature is sent in the X-Atlas-Signature header; the signing secret is rotated on request and the previous secret stays valid for one hour after rotation. A delivery that does not receive a 2xx response is retried five times with exponential backoff over a total window of six hours, after which the job result is available only by polling.

## Deprecation policy

A breaking change to an endpoint is announced at least 180 days before it takes effect, and the affected responses carry a Sunset header with the retirement date during that window. Additive changes, such as a new optional field, are shipped without notice, so clients must ignore unknown fields.
