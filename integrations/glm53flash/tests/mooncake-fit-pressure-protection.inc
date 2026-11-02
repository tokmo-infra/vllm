// Inserted only into the isolated native gtest translation unit.
TEST_F(MasterServiceTest, FitPressureProtectsLeasesPinsAndReclaimsEligible) {
    auto config = MasterServiceConfig::builder()
        .set_memory_allocator(BufferAllocatorType::OFFSET)
        .set_eviction_high_watermark_ratio(0.98)
        .set_eviction_ratio(0.05)
        .set_default_kv_lease_ttl(30000)
        .set_default_kv_soft_pin_ttl(30000)
        .set_allow_evict_soft_pinned_objects(false).build();
    MasterService service(config);
    PrepareSimpleSegment(service, "fit_protection_segment", 0x300000000, 32 * 1024 * 1024);
    PrepareSimpleSegment(service, "fit_copy_target", 0x400000000, 32 * 1024 * 1024);
    auto client = generate_uuid();
    auto put = [&](const std::string& key, ReplicateConfig rc) {
        return service.PutStart(client, key, TenantId::Default(), 1024 * 1024, rc).has_value() &&
               service.PutEnd(client, key, TenantId::Default(), ReplicaType::MEMORY).has_value();
    };
    ReplicateConfig ordinary; ordinary.replica_num = 1;
    ordinary.preferred_segment = "fit_protection_segment";
    ReplicateConfig hard = ordinary; hard.with_hard_pin = true;
    ReplicateConfig soft = ordinary; soft.soft_pin_action = SoftPinAction::ENABLE;
    ASSERT_TRUE(put("fit_hard", hard));
    ASSERT_TRUE(put("fit_soft", soft));
    ASSERT_TRUE(put("fit_lease", ordinary));
    auto lease = service.GetReplicaList("fit_lease", TenantId::Default());
    ASSERT_TRUE(lease.has_value());
    ReplicateConfig grouped = ordinary;
    grouped.group_ids = std::vector<std::string>{"fit_group"};
    ASSERT_TRUE(put("fit_group_a", grouped));
    ASSERT_TRUE(put("fit_group_b", grouped));
    ASSERT_TRUE(service.GetReplicaList("fit_group_a", TenantId::Default()).has_value());
    ASSERT_TRUE(put("fit_busy", ordinary));
    ASSERT_TRUE(service.CopyStart(client, "fit_busy", TenantId::Default(),
                                  "fit_protection_segment", {"fit_copy_target"}).has_value());
    ASSERT_TRUE(put("fit_victim", ordinary));
    const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(7);
    bool found = true;
    while (std::chrono::steady_clock::now() < until) {
        auto keys = service.GetAllKeys(TenantId::Default());
        ASSERT_TRUE(keys.has_value());
        found = std::find(keys->begin(), keys->end(), "fit_victim") != keys->end();
        if (!found) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    EXPECT_FALSE(found) << "fit-pressure did not exercise BatchEvict";
    EXPECT_TRUE(service.GetReplicaList("fit_hard", TenantId::Default()).has_value());
    EXPECT_TRUE(service.GetReplicaList("fit_soft", TenantId::Default()).has_value());
    EXPECT_TRUE(service.GetReplicaList("fit_lease", TenantId::Default()).has_value());
    EXPECT_TRUE(service.GetReplicaList("fit_group_a", TenantId::Default()).has_value());
    EXPECT_TRUE(service.GetReplicaList("fit_group_b", TenantId::Default()).has_value());
    EXPECT_TRUE(service.GetReplicaList("fit_busy", TenantId::Default()).has_value());
}
