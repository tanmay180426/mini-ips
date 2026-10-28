import sqlite3

connection = sqlite3.connect("data/mini_ids.db")

print("Total alert rows:")
print(connection.execute("SELECT COUNT(*) FROM alerts").fetchone()[0])

print("\nCounts by rule, severity, source, and destination:")
rows = connection.execute("""
    SELECT rule_name, severity, src_ip, dst_ip, COUNT(*)
    FROM alerts
    GROUP BY rule_name, severity, src_ip, dst_ip
    ORDER BY rule_name, severity, src_ip, dst_ip
""").fetchall()

for row in rows:
    print(row)

print("\nAlert timestamp range:")
print(connection.execute("""
    SELECT MIN(timestamp), MAX(timestamp)
    FROM alerts
""").fetchone())

print("\nPacket table total:")
print(connection.execute("SELECT COUNT(*) FROM packets").fetchone()[0])

connection.close()