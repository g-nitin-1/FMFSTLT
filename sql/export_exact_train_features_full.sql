WITH raw_measurements AS (
  SELECT
    m.date,
    m.uuid,
    m.speed_tier,
    m.test_time,
    m.mean_throughput_mbps AS y_true_mbps,
    sm.TCPInfo.ElapsedTime AS elapsed_us,
    sm.TCPInfo.BytesAcked AS bytes_acked,
    sm.TCPInfo.RTT AS rtt_us,
    sm.TCPInfo.SndCwnd AS snd_cwnd,
    sm.TCPInfo.SndMSS AS snd_mss,
    sm.TCPInfo.Unacked AS unacked,
    sm.TCPInfo.TotalRetrans AS total_retrans,
    sm.TCPInfo.DSackDups AS dsack_dups,
    sm.BBRInfo.BW AS bbr_bw
  FROM `ee-21cs01007.fmfstlt.paper_exact_train_800k_meta` AS m
  JOIN `measurement-lab.ndt_intermediate.extended_ndt7_downloads` AS t
    ON t.id = m.uuid
   AND t.date = m.date,
  UNNEST(t._internal202402.raw.Download.ServerMeasurements) AS sm
  WHERE sm.TCPInfo IS NOT NULL
    AND t.date IN (
      '2024-04-23', '2024-05-30', '2024-06-16', '2024-07-28',
      '2024-08-06', '2024-09-26', '2024-10-06', '2024-11-16',
      '2024-12-14', '2025-01-03'
    )
    AND sm.TCPInfo.ElapsedTime IS NOT NULL
    AND sm.TCPInfo.BytesAcked IS NOT NULL
    AND sm.TCPInfo.ElapsedTime > 0
    AND sm.TCPInfo.ElapsedTime < 10000000
),
annotated AS (
  SELECT
    date,
    uuid,
    speed_tier,
    test_time,
    y_true_mbps,
    elapsed_us,
    DIV(elapsed_us, 100000) AS bucket_100ms,
    COALESCE(
      8 * SAFE_DIVIDE(
        bytes_acked - LAG(bytes_acked) OVER (PARTITION BY uuid ORDER BY elapsed_us),
        elapsed_us - LAG(elapsed_us) OVER (PARTITION BY uuid ORDER BY elapsed_us)
      ),
      8 * SAFE_DIVIDE(bytes_acked, elapsed_us)
    ) AS inst_throughput_mbps,
    8 * SAFE_DIVIDE(bytes_acked, elapsed_us) AS cumavg_throughput_mbps,
    CAST(rtt_us AS FLOAT64) AS rtt_us,
    CAST(snd_cwnd AS FLOAT64) AS snd_cwnd,
    CAST(SAFE_MULTIPLY(CAST(unacked AS FLOAT64), CAST(snd_mss AS FLOAT64)) AS FLOAT64) AS bytes_in_flight,
    CAST(total_retrans AS FLOAT64) AS total_retrans,
    CAST(dsack_dups AS FLOAT64) AS dsack_dups,
    CAST(bbr_bw AS FLOAT64) AS bbr_bw,
    MAX(CAST(bbr_bw AS FLOAT64)) OVER (
      PARTITION BY uuid
      ORDER BY elapsed_us
      ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS prior_max_bbr_bw
  FROM raw_measurements
),
segmented AS (
  SELECT
    *,
    CASE
      WHEN prior_max_bbr_bw IS NULL THEN 1
      WHEN bbr_bw >= 1.25 * prior_max_bbr_bw THEN 1
      ELSE 0
    END AS growth_reset
  FROM annotated
),
per_sample AS (
  SELECT
    *,
    CAST(
      CASE
        WHEN ROW_NUMBER() OVER (
          PARTITION BY uuid, growth_segment
          ORDER BY elapsed_us
        ) >= 4 THEN 1
        ELSE 0
      END AS FLOAT64
    ) AS pipe_full_signal
  FROM (
    SELECT
      *,
      SUM(growth_reset) OVER (
        PARTITION BY uuid
        ORDER BY elapsed_us
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
      ) AS growth_segment
    FROM segmented
  )
),
bucketed AS (
  SELECT
    date,
    uuid,
    speed_tier,
    test_time,
    y_true_mbps,
    bucket_100ms,
    COUNT(*) AS measurement_count,
    AVG(inst_throughput_mbps) AS inst_throughput_mbps,
    AVG(cumavg_throughput_mbps) AS cumavg_throughput_mbps,
    SUM(pipe_full_signal) AS pipe_full_samples,
    AVG(rtt_us) AS mean_rtt_us,
    STDDEV_POP(rtt_us) AS std_rtt_us,
    AVG(snd_cwnd) AS mean_snd_cwnd,
    STDDEV_POP(snd_cwnd) AS std_snd_cwnd,
    AVG(bytes_in_flight) AS mean_bytes_in_flight,
    STDDEV_POP(bytes_in_flight) AS std_bytes_in_flight,
    AVG(total_retrans) AS mean_total_retrans,
    STDDEV_POP(total_retrans) AS std_total_retrans,
    AVG(dsack_dups) AS mean_dsack_dups,
    STDDEV_POP(dsack_dups) AS std_dsack_dups
  FROM per_sample
  GROUP BY date, uuid, speed_tier, test_time, y_true_mbps, bucket_100ms
)
SELECT
  date,
  uuid,
  speed_tier,
  test_time,
  y_true_mbps,
  bucket_100ms,
  measurement_count,
  inst_throughput_mbps,
  cumavg_throughput_mbps,
  pipe_full_samples,
  mean_rtt_us,
  std_rtt_us,
  mean_snd_cwnd,
  std_snd_cwnd,
  mean_bytes_in_flight,
  std_bytes_in_flight,
  mean_total_retrans,
  std_total_retrans,
  mean_dsack_dups,
  std_dsack_dups
FROM bucketed
ORDER BY date, speed_tier, uuid, bucket_100ms;
