import { createClient } from '@supabase/supabase-js'

const supabaseUrl = "https://ktludhxrewwbewkhcnqx.supabase.co"
const supabaseKey = "sb_publishable_nEF1O4jZZl5dfVEZMhJzGA_D3kXFvKC"

export const supabase = createClient(supabaseUrl, supabaseKey)
