if core.settings:has("fixed_map_seed") then
  math.randomseed(core.settings:get("fixed_map_seed"))
end

-- Reward for digging tree blocks; punish non-tree digs
minetest.register_on_dignode(function(pos, node)
 if string.find(node.name, "tree") then
   set_reward_once(15.0, 0.0)
 else
   set_reward_once(-5.0, 0.0)
 end
end)

minetest.register_globalstep(function(dtime)
  for _, player in ipairs(minetest.get_connected_players()) do
    local ctrl = player:get_player_control()

    if ctrl and ctrl.LMB then
      set_reward_once(0.5, 0.0)
      minetest.log("action", "[SustainedDig] Digging detected")
    end
  end
end)


-- Terminate if player dies
minetest.register_on_dieplayer(function(ObjectRef, reason)
 set_termination()
end)

-- On join: disable HUD and set midday
minetest.register_on_joinplayer(function(player, _last_login)
 minetest.set_timeofday(0.5)
 player:hud_set_flags({
   hotbar = false,
   crosshair = false,
   healthbar = false,
   chat = false,
 })
end)


